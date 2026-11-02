#!/usr/bin/env python3
"""Insertion campaign specification, process checks, coverage and summaries of the END/B rerun
(docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md, section 9: Part B). The counterpart of campaign_spec.py for the
I__ cells: coverage is derived from the declared 22-cell matrix, never from the rows that happened to arrive, and
every accepted process is checked against the inherited workload identity (permutation, initial set, per-batch ID
hashes of the original October 24 insertion logs) before any of its numbers is used.

Specification (JSON):
  name, salt                 campaign label; the schedule salt (informational)
  workers 64, cpus 0-63      ENV workers= / count= / cpus= every process must report
  population split70         POPULATION policy (S = floor(7 N / 10), H = N - S)
  processes 3, warm_rounds 2, measured_rounds 4, batches 3, method batch_sorted_in_timer, seed 20260928
  arms                       dfuds, dfuds-dt
  cells                      [{dataset, impl, pct}, ...]: the measured matrix (22 cells)
  datasets                   {id: {N, S, H, perm_hash, stored_sorted_hash}}: the split identity every process must reproduce
  identity                   {dataset: {pct: {b, final, ids_hash: [3 hashes]}}}: the per-size batch identity
  binaries                   {"bench_dictcmp": [sha16, ...]}: accepted executable identities (CELL_RUN binary_sha=)
  key_end                    optional {impl: policy}: the CONFIG line's key_end= value (end_leaf / end_leaf_front_and_rear)
  allow_missing_population_record   optional, fixture runs on logs that predate the POPULATION record only
  historical                 informational: where the identity values came from (root, binary, log hashes)

Headline per round: (T0 + T1 + T2) / (3 b) = the driver's INSERT_ROUND ns_per_key; per process the mean of the four
measured rounds; per cell the median and min / max of the three process means. Batch positions 0-2 are kept
separately (INSERT_BATCH ns_per_key, mean over the measured rounds per process, then median / min / max).

usage: insert_spec.py <insertion results root> <spec.json> <out-dir>
  root holds <dataset>/insert/{ledger.tsv, raw/I__*.log}. Writes insertion_process_obs.csv, insertion_aggregates.csv,
  insertion_batch_aggregates.csv, insertion_round_obs.csv, insertion_coverage.csv, insertion_violations.csv,
  insertion_identity.csv, insertion_report.md; prints INSERT_SPEC ...; exit 0 only when every cell is ACCEPTED.
"""
import collections
import csv
import hashlib
import json
import os
import re
import statistics
import sys

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
PCT_DIV = {'0.001': 100000, '0.01': 10000, '0.1': 1000, '1': 100, '10': 10}
PCTS = ['0.001', '0.01', '0.1', '1', '10']
DS_ORDER = ['c2_xml_recon', 'enwiki_20261001_ns0', 'webbase_2001', 'dna_full_31mers', 'uk_2014_urls']
LABEL = {'c2_xml_recon': 'XML', 'enwiki_20261001_ns0': 'Wikipedia', 'webbase_2001': 'WebBase', 'dna_full_31mers': 'DNA',
         'uk_2014_urls': 'UK-2014', 'dfuds': 'DFUDS', 'dfuds-dt': 'DT'}


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        s = json.load(f)
    for k in ('name', 'workers', 'cpus', 'population', 'processes', 'warm_rounds', 'measured_rounds', 'batches', 'method',
              'seed', 'arms', 'cells', 'datasets', 'identity', 'binaries'):
        if k not in s:
            raise ValueError(f'insertion spec {path}: missing field {k}')
    if s['population'] != 'split70':
        raise ValueError(f'insertion spec {path}: population must be split70 (the evolving-population protocol)')
    if s['batches'] != 3:
        raise ValueError(f'insertion spec {path}: batches must be 3')
    seen = set()
    for c in s['cells']:
        key = (c['dataset'], c['impl'], str(c['pct']))
        c['pct'] = str(c['pct'])
        if key in seen:
            raise ValueError(f'insertion spec {path}: duplicate cell {key}')
        seen.add(key)
        if c['impl'] not in s['arms'] or c['dataset'] not in s['datasets'] or c['pct'] not in PCT_DIV:
            raise ValueError(f'insertion spec {path}: cell {key} names an unknown arm, dataset or size')
        if c['pct'] not in s['identity'].get(c['dataset'], {}):
            raise ValueError(f'insertion spec {path}: cell {key} has no identity entry')
    for ds, d in s['datasets'].items():
        n = int(d['N'])
        if int(d['S']) != 7 * n // 10 or int(d['H']) != n - 7 * n // 10:
            raise ValueError(f'insertion spec {path}: {ds} S / H are not the 70 / 30 split of N')
        for pct, ident in s['identity'].get(ds, {}).items():
            if int(ident['b']) != n // PCT_DIV[pct] or int(ident['final']) != int(d['S']) + 3 * int(ident['b']):
                raise ValueError(f'insertion spec {path}: {ds} pct{pct} b / final do not follow N // divisor and S + 3 b')
            if len(ident['ids_hash']) != 3:
                raise ValueError(f'insertion spec {path}: {ds} pct{pct} needs three batch hashes')
    return s


def cells_of(spec):
    """The declared cells in the campaign's order (dataset order, then arm, then size)."""
    order = {d: i for i, d in enumerate(DS_ORDER)}
    return sorted(spec['cells'], key=lambda c: (order.get(c['dataset'], 99), spec['arms'].index(c['impl']), PCTS.index(c['pct'])))


def pcts_by_arm(spec, ds):
    out = {}
    for c in cells_of(spec):
        if c['dataset'] == ds:
            out.setdefault(c['impl'], []).append(c['pct'])
    return out


def parse_log(path):
    """One I__ log -> record."""
    r = {'path': path, 'cell_run': {}, 'command': '', 'env': {}, 'settings': {}, 'build': {}, 'config': {}, 'impl': {}, 'load': {},
         'split': {}, 'population': {}, 'plan': {}, 'ids': {}, 'builds': [], 'batches': [], 'rounds': [], 'memory_end': {},
         'status': '', 'check_fail': [], 'elapsed_s': '', 'max_rss_kb': ''}
    with open(path, errors='replace') as f:
        for line in f:
            tag = line.split(' ', 1)[0]
            if tag == 'CELL_RUN':
                r['cell_run'] = kv(line)
            elif tag == 'COMMAND':
                r['command'] = line.strip()
            elif tag == 'ENV':
                r['env'] = kv(line)
            elif tag == 'SETTINGS':
                r['settings'] = kv(line)
            elif tag == 'BUILD':
                r['build'] = kv(line)
            elif tag == 'CONFIG':
                parts = line.split(' ', 2)
                if len(parts) > 1:
                    r['config'][parts[1]] = kv(line)
            elif tag == 'IMPL':
                r['impl'] = kv(line)
            elif tag == 'LOAD':
                r['load'] = kv(line)
            elif tag == 'SPLIT':
                r['split'] = kv(line)
            elif tag == 'POPULATION':
                r['population'] = kv(line)
            elif tag == 'INSERT_PLAN':
                r['plan'] = kv(line)
            elif tag == 'INSERT_BATCH_IDS':
                d = kv(line)
                r['ids'][int(d['batch'])] = d['ids_hash']
            elif tag == 'INSERT_BUILD':
                r['builds'].append(kv(line))
            elif tag == 'INSERT_BATCH':
                r['batches'].append(kv(line))
            elif tag == 'INSERT_ROUND':
                r['rounds'].append(kv(line))
            elif tag == 'MEMORY_END':
                r['memory_end'] = kv(line)
            elif tag == 'CHECK_FAIL':
                r['check_fail'].append(line.strip())
            elif tag == 'CELL_END':
                r['status'] = kv(line).get('status', '')
            elif line.startswith('\tElapsed (wall clock) time'):
                t = line.strip().split()[-1]
                parts = [float(x) for x in t.split(':')]
                r['elapsed_s'] = parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]
            elif line.startswith('\tMaximum resident set size'):
                r['max_rss_kb'] = line.strip().split()[-1]
    return r


def check_process(spec, ds, impl, pct, p, r):
    """Violations of one accepted process against the specification and the inherited identity."""
    v = []
    if r['status'] != 'OK':
        v.append(f"status({r['status'] or 'missing'})")
    if r['check_fail']:
        v.append('check_fail')
    env = r['env']
    if env.get('workers') != str(spec['workers']) or env.get('count') != str(spec['workers']):
        v.append(f"workers({env.get('workers')},count={env.get('count')})")
    if spec.get('cpus') and env.get('cpus') != spec['cpus']:
        v.append(f"cpus({env.get('cpus')})")
    if r['settings'].get('seed') != str(spec['seed']):
        v.append(f"seed({r['settings'].get('seed')})")
    if r['settings'] and (r['settings'].get('warm_rounds') != str(spec['warm_rounds']) or r['settings'].get('rounds') != str(spec['measured_rounds'])):
        v.append(f"rounds_setting({r['settings'].get('warm_rounds')}+{r['settings'].get('rounds')})")
    d = spec['datasets'][ds]
    pol = r['population'].get('policy') or r['split'].get('policy')
    if pol != spec['population'] and not (pol is None and spec.get('allow_missing_population_record')):
        v.append(f'population_policy({pol})')
    sp = r['split']
    for k, want in (('N', d['N']), ('S', d['S']), ('H', d['H'])):
        if sp.get(k) != str(want):
            v.append(f"split_{k}({sp.get(k)})")
    for k in ('perm_hash', 'stored_sorted_hash'):
        if sp.get(k) != d[k]:
            v.append(f"{k}({sp.get(k)})")
    if sp.get('seed') != str(spec['seed']):
        v.append(f"split_seed({sp.get('seed')})")
    ident = spec['identity'][ds][pct]
    plan = r['plan']
    if not plan:
        v.append('no_insert_plan')
    else:
        if plan.get('pct') != pct:
            v.append(f"plan_pct({plan.get('pct')})")
        if plan.get('b') != str(ident['b']):
            v.append(f"plan_b({plan.get('b')})")
        if plan.get('batches') != str(spec['batches']):
            v.append(f"plan_batches({plan.get('batches')})")
        if plan.get('S') != str(d['S']) or plan.get('final') != str(ident['final']):
            v.append(f"plan_counts(S={plan.get('S')},final={plan.get('final')})")
        if plan.get('method') != spec['method']:
            v.append(f"method({plan.get('method')})")
    for t in range(3):
        if r['ids'].get(t) != ident['ids_hash'][t]:
            v.append(f"ids_hash_batch{t}({r['ids'].get(t)})")
    nb = spec['warm_rounds'] + spec['measured_rounds']
    if len(r['builds']) != nb:
        v.append(f'fresh_builds({len(r["builds"])})')
    mr = [x for x in r['rounds'] if x.get('measured') == '1']
    mb = [x for x in r['batches'] if x.get('measured') == '1']
    if len(mr) != spec['measured_rounds']:
        v.append(f'measured_rounds({len(mr)})')
    if len(mb) != spec['measured_rounds'] * spec['batches']:
        v.append(f'measured_batches({len(mb)})')
    if len(r['rounds']) != nb or len(r['batches']) != nb * spec['batches']:
        v.append(f'round_records({len(r["rounds"])}+{len(r["batches"])})')
    for x in r['batches']:
        if x.get('added') != x.get('b') or x.get('b') != str(ident['b']):
            v.append(f"batch_added(round{x.get('round')}:b{x.get('batch')}={x.get('added')}/{x.get('b')})")
            break
    for x in r['rounds']:
        if x.get('key_count') != str(ident['final']) or x.get('added') != str(3 * int(ident['b'])):
            v.append(f"round_count(round{x.get('round')}:key_count={x.get('key_count')},added={x.get('added')})")
            break
    run = r['cell_run']
    allowed = spec['binaries'].get('bench_dictcmp', [])
    if allowed and run.get('binary_sha') not in allowed:
        v.append(f"binary({run.get('binary_sha')})")
    ke = (spec.get('key_end') or {}).get(impl)
    if ke:
        got = next((c.get('key_end') for c in r['config'].values() if 'key_end' in c), None)
        if got != ke:
            v.append(f'key_end({got})')
    if r['impl'] and r['impl'].get('name') != impl:
        v.append(f"impl({r['impl'].get('name')})")
    hf = r['memory_end'].get('huge_frac')
    if hf is None:
        v.append('no_memory_record')
    elif float(hf) > 0.01:
        v.append(f'huge_pages({hf})')
    return sorted(set(v))


def collect(root, spec):
    """Accepted-by-ledger processes of every dataset under <root>/<dataset>/insert, checked against the specification.
    Returns (accepted, violations, rejected, ledger_rows) keyed by (dataset, impl, pct, process)."""
    accepted, violations, rejected, ledger = {}, {}, {}, []
    for ds in spec['datasets']:
        led = os.path.join(root, ds, 'insert', 'ledger.tsv')
        if not os.path.exists(led):
            continue
        rows = list(csv.DictReader(open(led), delimiter='\t'))
        for row in rows:
            row['dataset'] = ds
            ledger.append(row)
        last = {}
        for row in rows:
            last[row['name']] = row
        for name, row in last.items():
            m = re.fullmatch(r'I__(.+)__(.+)__pct([0-9.]+)__p(\d+)', name)
            if not m or m.group(1) != ds:
                continue
            key = (ds, m.group(2), m.group(3), int(m.group(4)))
            if key[1] not in spec['arms'] or key[2] not in spec['identity'].get(ds, {}):
                continue
            if not any(c['dataset'] == ds and c['impl'] == key[1] and c['pct'] == key[2] for c in spec['cells']):
                continue   # a row outside the declared matrix is never summarized (coverage is declared, not discovered)
            log = os.path.join(root, ds, 'insert', 'raw', name + '.log')
            if row['verdict'] != 'ACCEPTED' or not os.path.exists(log):
                rejected[key] = row.get('reasons', '')
                continue
            r = parse_log(log)
            r['ledger'] = row
            r['log_sha256'] = sha256(log)
            v = check_process(spec, *key, r)
            if v:
                violations[key] = v
            else:
                accepted[key] = r
    return accepted, violations, rejected, ledger


def coverage(spec, accepted, violations, rejected):
    rows = []
    for c in cells_of(spec):
        ds, impl, pct = c['dataset'], c['impl'], c['pct']
        have = [p for p in range(1, spec['processes'] + 1) if (ds, impl, pct, p) in accepted]
        bad = [p for p in range(1, spec['processes'] + 1) if (ds, impl, pct, p) in violations]
        rej = [p for p in range(1, spec['processes'] + 1) if (ds, impl, pct, p) in rejected and p not in have]
        status = ('ACCEPTED' if len(have) >= spec['processes'] else 'PARTIAL' if have else 'SPEC_VIOLATION' if bad
                  else 'REJECTED' if rej else 'MISSING')
        rows.append({'dataset': ds, 'impl': impl, 'pct': pct, 'b': spec['identity'][ds][pct]['b'], 'status': status,
                     'accepted_processes': ''.join(str(p) for p in have), 'rejected_processes': ''.join(str(p) for p in rej),
                     'note': '; '.join(f"p{p}: {','.join(violations[(ds, impl, pct, p)])}" for p in bad)})
    return rows


def summarize(spec, accepted):
    """Process observations and the cell / batch-position aggregates."""
    obs, rounds_out = [], []
    for (ds, impl, pct, p), r in sorted(accepted.items()):
        mr = [x for x in r['rounds'] if x.get('measured') == '1']
        mb = [x for x in r['batches'] if x.get('measured') == '1']
        means = statistics.mean(float(x['ns_per_key']) for x in mr)
        per_batch = {t: statistics.mean(float(x['ns_per_key']) for x in mb if int(x['batch']) == t) for t in range(3)}
        sort_share = statistics.mean(float(x['sort_ns']) / float(x['ns']) for x in mb if float(x['ns']) > 0)
        builds = [float(x['build_ns']) / 1e9 for x in r['builds']]
        o = {'dataset': ds, 'impl': impl, 'pct': pct, 'process': p, 'b': r['plan']['b'], 'S': r['plan']['S'], 'final': r['plan']['final'],
             'ns_per_key_mean': means, 'batch0_ns_per_key': per_batch[0], 'batch1_ns_per_key': per_batch[1], 'batch2_ns_per_key': per_batch[2],
             'sort_share_mean': sort_share, 'measured_rounds': len(mr), 'fresh_builds': len(r['builds']), 'build_s_mean': statistics.mean(builds),
             'elapsed_s': r['elapsed_s'], 'max_rss_kb': r['max_rss_kb'], 'binary_sha16': r['cell_run'].get('binary_sha'),
             'ledger_ident': r['ledger']['ident'], 'attempt': r['ledger']['attempt'], 'start': r['ledger']['start'], 'end': r['ledger']['end'],
             'perm_hash': r['split'].get('perm_hash'), 'stored_sorted_hash': r['split'].get('stored_sorted_hash'),
             'ids_hash': ';'.join(r['ids'].get(t, '') for t in range(3)), 'log': os.path.relpath(r['path'], os.path.dirname(os.path.dirname(os.path.dirname(r['path'])))),
             'log_sha256': r['log_sha256']}
        obs.append(o)
        for x in mr:
            rounds_out.append({'dataset': ds, 'impl': impl, 'pct': pct, 'process': p, 'round': x['round'], 'ns': x['ns'], 'added': x['added'],
                               'ns_per_key': x['ns_per_key'], 'key_count': x['key_count'], 'teardown_ns': x.get('teardown_ns', '')})
        for x in mb:
            rounds_out.append({'dataset': ds, 'impl': impl, 'pct': pct, 'process': p, 'round': x['round'], 'batch': x['batch'], 'ns': x['ns'],
                               'added': x['added'], 'ns_per_key': x['ns_per_key'], 'sort_ns': x['sort_ns']})
    aggs, baggs = [], []
    for c in cells_of(spec):
        ds, impl, pct = c['dataset'], c['impl'], c['pct']
        po = [o for o in obs if (o['dataset'], o['impl'], o['pct']) == (ds, impl, pct)]
        vals = [o['ns_per_key_mean'] for o in po]
        a = {'dataset': ds, 'impl': impl, 'pct': pct, 'b': spec['identity'][ds][pct]['b'], 'n_processes': len(vals),
             'median_ns_per_key': statistics.median(vals) if vals else '', 'min_ns_per_key': min(vals) if vals else '', 'max_ns_per_key': max(vals) if vals else '',
             'process_means': ';'.join(f'{v:.4f}' for v in vals), 'sort_share_mean': statistics.mean(o['sort_share_mean'] for o in po) if po else '',
             'build_s_mean': statistics.mean(o['build_s_mean'] for o in po) if po else '',
             'avg_batch_ms': statistics.median(vals) * int(spec['identity'][ds][pct]['b']) / 1e6 if vals else '',
             'binary_sha16': ','.join(sorted({o['binary_sha16'] or '' for o in po})), 'status': 'ACCEPTED' if len(vals) == spec['processes'] else ('PARTIAL' if vals else 'MISSING')}
        aggs.append(a)
        for t in range(3):
            bv = [o[f'batch{t}_ns_per_key'] for o in po]
            baggs.append({'dataset': ds, 'impl': impl, 'pct': pct, 'batch': t, 'n_processes': len(bv), 'median_ns_per_key': statistics.median(bv) if bv else '',
                          'min_ns_per_key': min(bv) if bv else '', 'max_ns_per_key': max(bv) if bv else '', 'process_means': ';'.join(f'{v:.4f}' for v in bv)})
    return obs, rounds_out, aggs, baggs


def write_csv(path, cols, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fmt(v, nd=2):
    try:
        return f'{float(v):,.{nd}f}'
    except (TypeError, ValueError):
        return '-'


def agg_text(a):
    if not a or a['n_processes'] == 0:
        return 'MISSING'
    s = f"{fmt(a['median_ns_per_key'])} [{fmt(a['min_ns_per_key'])}-{fmt(a['max_ns_per_key'])}]"
    return s + (f" (n={a['n_processes']})" if a['n_processes'] != 3 else '')


def report(spec, out, aggs, baggs, cov, violations):
    A = {(a['dataset'], a['impl'], a['pct']): a for a in aggs}
    B = {(a['dataset'], a['impl'], a['pct'], a['batch']): a for a in baggs}
    L = [f"# Insertion rerun ({spec['name']}): tables\n",
         'Protocol: 70 % initial population (S = floor(7 N / 10)), held-out order defines the insertion sequence, three '
         'consecutive disjoint batches of b = N // divisor keys per round, a fresh S-key build per round outside the timer, '
         f"{spec['warm_rounds']} warm-up + {spec['measured_rounds']} measured rounds per process, {spec['processes']} independent processes "
         f"per cell, {spec['workers']} workers. Headline = (T0 + T1 + T2) / (3 b) ns per inserted key (sort inside the timer), mean over the "
         'measured rounds per process, then median [min-max] of the process means. Average batch duration = ns/key x b, not the sum of '
         'the three batches.\n']
    wb = 'webbase_2001'
    if any(c['dataset'] == wb for c in spec['cells']):
        L.append('## WebBase insertion sweep (Figure 1): ns per inserted key\n')
        L.append('| Arm | ' + ' | '.join(f'{p} % (b = {spec["identity"][wb][p]["b"]:,})' for p in PCTS if (wb, 'dfuds', p) in A or (wb, 'dfuds-dt', p) in A) + ' |')
        ps = [p for p in PCTS if (wb, 'dfuds', p) in A or (wb, 'dfuds-dt', p) in A]
        L.append('|---|' + '---:|' * len(ps))
        for impl in spec['arms']:
            L.append(f'| {LABEL[impl]} | ' + ' | '.join(agg_text(A.get((wb, impl, p))) for p in ps) + ' |')
        L.append('')
        L.append('Batch positions (ns per inserted key, median [min-max] of the process means of that batch):\n')
        L.append('| Arm | size | batch 1 | batch 2 | batch 3 | all three |')
        L.append('|---|---|---:|---:|---:|---:|')
        for impl in spec['arms']:
            for p in ps:
                if (wb, impl, p) in A:
                    L.append(f'| {LABEL[impl]} | {p} % | ' + ' | '.join(agg_text(B.get((wb, impl, p, t))) for t in range(3)) + f' | {agg_text(A.get((wb, impl, p)))} |')
        L.append('')
    L.append('## Five-dataset 10 % insertion (Table 3): ns per inserted key\n')
    L.append('| Dataset | b | ' + ' | '.join(LABEL[i] for i in spec['arms']) + ' |')
    L.append('|---|---:|' + '---:|' * len(spec['arms']))
    for ds in DS_ORDER:
        if ds in spec['datasets'] and any((ds, i, '10') in A for i in spec['arms']):
            L.append(f"| {LABEL[ds]} | {spec['identity'][ds]['10']['b']:,} | " + ' | '.join(agg_text(A.get((ds, i, '10'))) for i in spec['arms']) + ' |')
    L.append('')
    L.append('## DFUDS 0.001 % endpoints (section 6.2 prose): ns per inserted key\n')
    L.append('| Dataset | b | DFUDS 0.001 % | DFUDS 10 % | ratio 0.001 % / 10 % |')
    L.append('|---|---:|---:|---:|---:|')
    for ds in DS_ORDER:
        if (ds, 'dfuds', '0.001') in A:
            a, b = A[(ds, 'dfuds', '0.001')], A.get((ds, 'dfuds', '10'))
            ratio = fmt(float(a['median_ns_per_key']) / float(b['median_ns_per_key']), 1) if a['n_processes'] and b and b['n_processes'] else '-'
            L.append(f"| {LABEL[ds]} | {spec['identity'][ds]['0.001']['b']:,} | {agg_text(a)} | {agg_text(b)} | {ratio} |")
    L.append('')
    L.append('Non-WebBase endpoints are single sizes, not a sweep: the plan does not measure 0.01-1 % or the DT small batches there.\n')
    L.append('## Coverage\n')
    cnt = collections.Counter(c['status'] for c in cov)
    L.append('Status counts: ' + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items())) + f" (declared cells {len(cov)}, processes {len(cov) * spec['processes']}).\n")
    L.append('| Dataset | Arm | size | b | status | accepted | note |')
    L.append('|---|---|---|---:|---|---|---|')
    for c in cov:
        L.append(f"| {LABEL[c['dataset']]} | {LABEL[c['impl']]} | {c['pct']} % | {int(c['b']):,} | {c['status']} | {c['accepted_processes']} | {c['note']} |")
    L.append('')
    if violations:
        L.append('## Specification violations (processes excluded from every aggregate)\n')
        for k, v in sorted(violations.items()):
            L.append(f'- {k[0]} {k[1]} pct{k[2]} p{k[3]}: {", ".join(v)}')
        L.append('')
    open(os.path.join(out, 'insertion_report.md'), 'w').write('\n'.join(L) + '\n')


def main():
    root, spec_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = load(spec_path)
    os.makedirs(out, exist_ok=True)
    accepted, violations, rejected, ledger = collect(root, spec)
    cov = coverage(spec, accepted, violations, rejected)
    obs, rounds_out, aggs, baggs = summarize(spec, accepted)
    write_csv(os.path.join(out, 'insertion_process_obs.csv'), list(obs[0].keys()) if obs else ['dataset'], obs)
    write_csv(os.path.join(out, 'insertion_round_obs.csv'), ['dataset', 'impl', 'pct', 'process', 'round', 'batch', 'ns', 'added', 'ns_per_key', 'sort_ns', 'key_count', 'teardown_ns'], rounds_out)
    write_csv(os.path.join(out, 'insertion_aggregates.csv'), list(aggs[0].keys()) if aggs else ['dataset'], aggs)
    write_csv(os.path.join(out, 'insertion_batch_aggregates.csv'), list(baggs[0].keys()) if baggs else ['dataset'], baggs)
    write_csv(os.path.join(out, 'insertion_coverage.csv'), ['dataset', 'impl', 'pct', 'b', 'status', 'accepted_processes', 'rejected_processes', 'note'], cov)
    write_csv(os.path.join(out, 'insertion_violations.csv'), ['dataset', 'impl', 'pct', 'process', 'violations'],
              [{'dataset': k[0], 'impl': k[1], 'pct': k[2], 'process': k[3], 'violations': ';'.join(v)} for k, v in sorted(violations.items())])
    write_csv(os.path.join(out, 'insertion_identity.csv'), ['dataset', 'impl', 'pct', 'process', 'perm_hash', 'stored_sorted_hash', 'ids_hash', 'S', 'final', 'binary_sha16', 'log_sha256'], obs)
    write_csv(os.path.join(out, 'insertion_attempts.csv'), ['dataset', 'name', 'kind', 'ident', 'attempt', 'rc', 'start', 'end', 'verdict', 'reasons'], ledger)
    with open(os.path.join(out, 'insertion_spec.json'), 'w') as f:
        json.dump(spec, f, indent=1)
    report(spec, out, aggs, baggs, cov, violations)
    n_acc = sum(1 for c in cov if c['status'] == 'ACCEPTED')
    complete = n_acc == len(cov)
    print(f"INSERT_SPEC spec={spec['name']} accepted_processes={len(accepted)} violations={len(violations)} rejected={len(rejected)} "
          f"coverage_accepted={n_acc}/{len(cov)} cells complete={int(complete)} out={out}")
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
