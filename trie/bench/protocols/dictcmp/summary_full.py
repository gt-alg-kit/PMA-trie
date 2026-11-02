#!/usr/bin/env python3
"""Summaries of the full-population, hit-only round (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md,
sections 5, 6 and 9). Reads every <results-root>/<dataset>/sessions/raw/S__<dataset>__<arm>__p<k>.log with its ledger
row and writes into <out-dir>:
  attempts.csv        every ledger row (accepted, rejected, interfered attempts)
  build_obs.csv       one row per accepted process: the single end-to-end build (ns, sort_ns, construct_ns), memory
  build_summary.csv   per dataset/arm: n, median / min / max of the three single-build observations (e2e, construct, sort)
  point_rounds.csv    every MEASURED point round of the measure phase (calibration / profile rounds excluded)
  point_batches.csv   every measured point batch (elapsed ns, batch size)
  point_summary.csv   per dataset/arm/size/mode: the process means (mean of 4 measured rounds), median / min / max over
                      the accepted processes, L, bq, round duration
  range_rounds.csv / range_batches.csv / range_summary.csv   the same for the range cell (ns/request primary;
                      keys/request, bytes/request, truncated/request, ns/returned key supplementary)
  space.csv / space_components.csv   the process-1 census of the retained index (standalone bytes, B/key, B per input key byte)
  validate.csv        every VALIDATE check of every process
  calibration.csv     every CALIBRATION_END (the frozen L and its source)
  profile.csv         PROFILE_END rows: control_before / profiled / control_after wall ns per profiled range cell
  coverage.csv        the expected matrix (build, census, every point size x mode, range x mode) with the status:
                      accepted processes, rejected, UNSUPPORTED_API (no scanner / no shared traversal), SIZE_LIMIT /
                      UNSUPPORTED_KEY_DOMAIN / RESOURCE_LIMIT (CELL_END status), MISSING
  figures/            build_time.png, serving_space.png, point_10pct.png, point_uk_sweep.png, range_k100.png (if matplotlib)
Aggregation (plan section 9): query cell = mean of the four measured rounds per process, then median and min / max of
the process means (whiskers = that range, not a confidence interval); build = median / min / max of the three single
observations; space = the labelled process-1 record.
usage: summary_full.py <results-root> <out-dir>
"""
import csv
import json
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # campaign_spec.py next to this script

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
ARMS = ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-64', 'marisa', 'pdt-static', 'dynpdt-compact']
SHARED = {'dfuds', 'dfuds-dt'}
RANGE = {'dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-64'}
DATASETS = ['c2_xml_recon', 'enwiki_20261001_ns0', 'webbase_2001', 'dna_full_31mers', 'uk_2014_urls']
LABEL = {'c2_xml_recon': 'XML', 'enwiki_20261001_ns0': 'Wikipedia', 'webbase_2001': 'WebBase', 'dna_full_31mers': 'DNA',
         'uk_2014_urls': 'UK-2014', 'dfuds': 'DFUDS', 'dfuds-dt': 'DT', 'art-olc': 'ART-OLC', 'hot-rowex': 'HOT-ROWEX',
         'fst-64': "FST'", 'marisa': 'MARISA', 'pdt-static': 'PDT', 'dynpdt-compact': 'PDT-Dyn'}
UK_SIZES = ['pct0.001', 'pct0.01', 'pct0.1', 'pct1', 'pct10']


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def med3(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ('', '', '', 0)
    return (statistics.median(xs), min(xs), max(xs), len(xs))


def parse_log(path):
    """One session log -> dict of records."""
    r = {'build': None, 'point_rounds': [], 'point_batches': [], 'range_rounds': [], 'range_batches': [], 'validate': [],
         'calibration': [], 'profile': [], 'space': None, 'components': [], 'status': '', 'memory_end': None, 'load': None,
         'cells_seen': set(), 'info': None, 'perf_record': [], 'plan': {}, 'paired': None, 'env': {}, 'cell_run': {},
         'population': {}, 'command': '', 'path': path, 'config': {}, 'session_summary': {}}
    phase = 'pre'
    cur_point = None
    cur_range = None
    cell_key = ''
    with open(path, errors='replace') as f:
        for line in f:
            tag = line.split(' ', 1)[0]
            if tag == 'ENV':
                r['env'] = kv(line)
            elif tag == 'CELL_RUN':
                r['cell_run'] = kv(line)
            elif tag == 'COMMAND':
                r['command'] = line.strip()
            elif tag == 'POPULATION':
                r['population'] = kv(line)
            elif tag == 'CONFIG':   # CONFIG <NAME> k=v ...: the arm's compiled configuration (key_end= since the END/B integration)
                parts = line.split(' ', 2)
                if len(parts) > 1:
                    r['config'][parts[1]] = kv(line)
            elif tag == 'SESSION_SUMMARY':
                r['session_summary'] = kv(line)
            elif tag == 'SESSION_PHASE':
                phase = kv(line)['phase']
            elif tag == 'SESSION_PLAN':
                r['plan'] = kv(line)
            elif tag == 'RANGE_PAIRED_TRACE':
                r['paired'] = kv(line)
            elif tag == 'SESSION_CELL':
                cell_key = kv(line)['key']
            elif tag == 'SESSION_BUILD':
                r['build'] = kv(line)
            elif tag == 'LOAD':
                r['load'] = kv(line)
            elif tag == 'POINT_CELL':
                d = kv(line)
                if 'status' in d:
                    continue
                cur_point = d
            elif tag == 'RANGE_CELL':
                d = kv(line)
                if 'status' in d:
                    continue
                cur_range = d
            elif tag == 'POINT_ROUND' and phase == 'measure':
                d = kv(line)
                if d.get('measured') == '1' and cur_point:
                    d.update({'bq': cur_point['bq'], 'size': cur_point.get('bq_label', ''), 'L': cur_point['L'], 'key': cell_key,
                              'mode': d.get('mode', cur_point.get('mode', ''))})
                    r['point_rounds'].append(d)
                    r['cells_seen'].add(cell_key)
            elif tag == 'POINT_BATCH' and phase == 'measure':
                d = kv(line)
                if d.get('measured') == '1' and cur_point:
                    d.update({'size': cur_point.get('bq_label', ''), 'key': cell_key})
                    r['point_batches'].append(d)
            elif tag == 'RANGE_ROUND' and phase == 'measure':
                d = kv(line)
                if d.get('measured') == '1' and cur_range:
                    d.update({'L': cur_range['L'], 'key': cell_key, 'paired': cur_range.get('paired', '0'), 'workload': cur_range.get('workload', ''),
                              'starts_ids_hash': cur_range.get('starts_ids_hash', ''), 'starts_bytes_hash': cur_range.get('starts_bytes_hash', '')})
                    r['range_rounds'].append(d)
                    r['cells_seen'].add(f"range:{d['k']}:{d['mode']}")
            elif tag == 'RANGE_BATCH' and phase == 'measure':
                d = kv(line)
                if d.get('measured') == '1':
                    d['key'] = cell_key
                    r['range_batches'].append(d)
            elif tag == 'VALIDATE':
                d = kv(line)
                d['verdict'] = line.strip().split()[-1]
                r['validate'].append(d)
            elif tag == 'CALIBRATION_END':
                r['calibration'].append(kv(line))
            elif tag == 'PROFILE_END':
                r['profile'].append(kv(line))
            elif tag == 'PROFILE_RECORD':
                r['perf_record'].append(kv(line))
            elif tag == 'SPACE' and 'checkpoint=session' in line:
                r['space'] = kv(line)
            elif tag == 'SPACE_COMPONENT' and 'checkpoint=session' in line:
                r['components'].append(kv(line))
            elif tag == 'INFO' and 'checkpoint=session' in line:
                r['info'] = line.strip()
            elif tag == 'MEMORY_END':
                r['memory_end'] = kv(line)
            elif tag == 'CELL_END':
                r['status'] = kv(line).get('status', '')
    return r


def write_csv(path, cols, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    root, out = sys.argv[1], sys.argv[2]
    spec = None
    if '--spec' in sys.argv:   # the declared campaign (campaign_spec.py): coverage and the process checks follow it
        import campaign_spec
        spec = campaign_spec.load(sys.argv[sys.argv.index('--spec') + 1])
    os.makedirs(out, exist_ok=True)
    attempts, build_obs, prounds, pbatches, rrounds, rbatches, validate, calib, profile, space, comps = ([] for _ in range(11))
    accepted = {}   # (ds, impl, p) -> parsed log
    statuses = {}   # (ds, impl, p) -> CELL_END status of the last attempt
    rejected = {}   # (ds, impl, p) -> reasons
    for ds in DATASETS:
        if spec and ds not in spec['datasets']:
            continue
        led = os.path.join(root, ds, 'sessions', 'ledger.tsv')
        if not os.path.exists(led):
            continue
        with open(led) as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        for row in rows:
            row['dataset'] = ds
            attempts.append(row)
        last = {}
        for row in rows:
            last[row['name']] = row
        for name, row in last.items():
            m = re.fullmatch(r'S__(.+)__(.+)__p(\d+)', name)
            if not m:
                continue
            key = (ds, m.group(2), int(m.group(3)))
            if spec and m.group(2) not in spec['arms']:
                continue
            log = os.path.join(root, ds, 'sessions', 'raw', name + '.log')
            if not os.path.exists(log):
                continue
            parsed = parse_log(log)
            statuses[key] = parsed['status']
            if row['verdict'] == 'ACCEPTED' and parsed['status'] == 'OK':
                accepted[key] = parsed
            else:
                rejected[key] = row['reasons']
    # ---- the specification's process checks: a violating process leaves every aggregate (plan section 5)
    violations = {}
    if spec:
        import campaign_spec
        for key, r in list(accepted.items()):
            v = campaign_spec.check_process(spec, key[0], key[1], key[2], r)
            if v:
                violations[key] = v
        for ds in spec['datasets']:
            for key, v in campaign_spec.trace_violations(spec, ds, {k: r for k, r in accepted.items() if k[0] == ds}).items():
                violations.setdefault(key, []).extend(v)
        for key in violations:
            accepted.pop(key, None)
        write_csv(os.path.join(out, 'spec_violations.csv'), ['dataset', 'impl', 'process', 'violations'],
                  [{'dataset': k[0], 'impl': k[1], 'process': k[2], 'violations': ';'.join(v)} for k, v in sorted(violations.items())])
        with open(os.path.join(out, 'campaign_spec.json'), 'w') as f:
            json.dump(spec, f, indent=1)
    # ---- builds
    for (ds, impl, p), r in sorted(accepted.items()):
        b = r['build']
        if not b:
            continue
        build_obs.append({'dataset': ds, 'impl': impl, 'process': p, 'e2e_s': float(b['ns']) / 1e9, 'sort_s': float(b['sort_ns']) / 1e9,
                          'construct_s': float(b['construct_ns']) / 1e9, 'key_count': b['key_count'], 'hwm_gb': float(b['hwm_kb']) / 1048576,
                          'peak_delta_gb': float(b['peak_delta_kb']) / 1048576, 'rss_before_gb': float(b['rss_before_kb']) / 1048576,
                          'input_order': b['input_order'], 'parallelism': b['parallelism'], 'contract': b['contract'],
                          'process_hwm_gb': float(r['memory_end']['hwm_kb']) / 1048576 if r['memory_end'] else ''})
    write_csv(os.path.join(out, 'build_obs.csv'), ['dataset', 'impl', 'process', 'contract', 'e2e_s', 'sort_s', 'construct_s', 'key_count',
                                                   'hwm_gb', 'peak_delta_gb', 'rss_before_gb', 'process_hwm_gb', 'input_order', 'parallelism'], build_obs)
    bsum = []
    for ds in DATASETS:
        for impl in ARMS:
            obs = [o for o in build_obs if o['dataset'] == ds and o['impl'] == impl]
            if not obs:
                continue
            e = med3([o['e2e_s'] for o in obs]); c = med3([o['construct_s'] for o in obs]); s = med3([o['sort_s'] for o in obs])
            bsum.append({'dataset': ds, 'impl': impl, 'n': len(obs), 'e2e_median_s': e[0], 'e2e_min_s': e[1], 'e2e_max_s': e[2],
                         'construct_median_s': c[0], 'construct_min_s': c[1], 'construct_max_s': c[2], 'sort_median_s': s[0],
                         'sort_min_s': s[1], 'sort_max_s': s[2], 'peak_delta_gb_max': max(o['peak_delta_gb'] for o in obs),
                         'input_order': obs[0]['input_order'], 'parallelism': obs[0]['parallelism']})
    write_csv(os.path.join(out, 'build_summary.csv'), list(bsum[0].keys()) if bsum else ['dataset', 'impl', 'n'], bsum)
    # ---- point
    for (ds, impl, p), r in sorted(accepted.items()):
        for d in r['point_rounds']:
            prounds.append({'dataset': ds, 'impl': impl, 'process': p, 'size': d['size'], 'bq': d['bq'], 'mode': d['mode'], 'L': d['L'],
                            'round': d['round'], 'ns': d['ns'], 'queries': d['queries'], 'ns_per_query': d['ns_per_query']})
        for d in r['point_batches']:
            pbatches.append({'dataset': ds, 'impl': impl, 'process': p, 'size': d['size'], 'mode': d['mode'], 'round': d['round'],
                             'batch': d['batch'], 'bq': d['bq'], 'ns': d['ns'], 'hits': d['hits']})
    write_csv(os.path.join(out, 'point_rounds.csv'), ['dataset', 'impl', 'process', 'size', 'bq', 'mode', 'L', 'round', 'ns', 'queries', 'ns_per_query'], prounds)
    write_csv(os.path.join(out, 'point_batches.csv'), ['dataset', 'impl', 'process', 'size', 'mode', 'round', 'batch', 'bq', 'ns', 'hits'], pbatches)
    psum = []
    groups = {}
    for d in prounds:
        groups.setdefault((d['dataset'], d['impl'], d['size'], d['mode']), {}).setdefault(d['process'], []).append(d)
    for (ds, impl, size, mode), procs in sorted(groups.items()):
        means = {p: statistics.mean(float(x['ns_per_query']) for x in v) for p, v in procs.items()}
        m = med3(list(means.values()))
        any_round = next(iter(procs.values()))[0]
        rounds_ms = [float(x['ns']) / 1e6 for v in procs.values() for x in v]
        psum.append({'dataset': ds, 'impl': impl, 'size': size, 'bq': any_round['bq'], 'mode': mode, 'L': any_round['L'],
                     'n_processes': len(means), 'rounds_per_process': {p: len(v) for p, v in procs.items()}.__repr__().replace(',', ';'),
                     'p1_ns_per_query': means.get(1, ''), 'p2_ns_per_query': means.get(2, ''), 'p3_ns_per_query': means.get(3, ''),
                     'median_ns_per_query': m[0], 'min_ns_per_query': m[1], 'max_ns_per_query': m[2],
                     'round_ms_median': statistics.median(rounds_ms), 'short_round': int(statistics.median(rounds_ms) < 100)})
    write_csv(os.path.join(out, 'point_summary.csv'), list(psum[0].keys()) if psum else ['dataset'], psum)
    # ---- range
    for (ds, impl, p), r in sorted(accepted.items()):
        for d in r['range_rounds']:
            rrounds.append({'dataset': ds, 'impl': impl, 'process': p, 'k': d['k'], 'mode': d['mode'], 'L': d['L'], 'br': d['br'], 'round': d['round'],
                            'ns': d['ns'], 'requests': d['requests'], 'keys': d['keys'], 'bytes': d['bytes'], 'ns_per_request': d['ns_per_request'],
                            'ns_per_key': d['ns_per_key'], 'paired': d['paired'], 'workload': d['workload'], 'starts_ids_hash': d['starts_ids_hash'],
                            'starts_bytes_hash': d['starts_bytes_hash']})
        for d in r['range_batches']:
            rbatches.append({'dataset': ds, 'impl': impl, 'process': p, 'k': d['k'], 'mode': d['mode'], 'round': d['round'], 'batch': d['batch'],
                             'br': d['br'], 'ns': d['ns'], 'keys': d['keys'], 'bytes': d['bytes'], 'empty': d['empty'], 'truncated': d['truncated']})
    write_csv(os.path.join(out, 'range_rounds.csv'), ['dataset', 'impl', 'process', 'k', 'mode', 'L', 'br', 'round', 'ns', 'requests', 'keys', 'bytes',
                                                      'ns_per_request', 'ns_per_key', 'paired', 'workload', 'starts_ids_hash', 'starts_bytes_hash'], rrounds)
    write_csv(os.path.join(out, 'range_batches.csv'), ['dataset', 'impl', 'process', 'k', 'mode', 'round', 'batch', 'br', 'ns', 'keys', 'bytes', 'empty', 'truncated'], rbatches)
    rsum = []
    groups = {}
    for d in rrounds:
        groups.setdefault((d['dataset'], d['impl'], d['k'], d['mode']), {}).setdefault(d['process'], []).append(d)
    for (ds, impl, k, mode), procs in sorted(groups.items()):
        means = {p: statistics.mean(float(x['ns_per_request']) for x in v) for p, v in procs.items()}
        m = med3(list(means.values()))
        allr = [x for v in procs.values() for x in v]
        req = sum(float(x['requests']) for x in allr); keys = sum(float(x['keys']) for x in allr); byt = sum(float(x['bytes']) for x in allr)
        bt = [b for b in rbatches if b['dataset'] == ds and b['impl'] == impl and b['k'] == k and b['mode'] == mode]
        trunc = sum(int(b['truncated']) for b in bt); breq = sum(int(b['br']) for b in bt)
        rsum.append({'dataset': ds, 'impl': impl, 'k': k, 'mode': mode, 'L': allr[0]['L'], 'br': allr[0]['br'], 'n_processes': len(means),
                     'paired': allr[0]['paired'], 'workload': allr[0]['workload'], 'starts_ids_hash': allr[0]['starts_ids_hash'],
                     'p1_ns_per_request': means.get(1, ''), 'p2_ns_per_request': means.get(2, ''), 'p3_ns_per_request': means.get(3, ''),
                     'median_ns_per_request': m[0], 'min_ns_per_request': m[1], 'max_ns_per_request': m[2],
                     'keys_per_request': keys / req if req else '', 'bytes_per_request': byt / req if req else '',
                     'truncated_per_request': trunc / breq if breq else '', 'ns_per_returned_key_median': m[0] * req / keys if keys and m[0] != '' else '',
                     'round_ms_median': statistics.median(float(x['ns']) / 1e6 for x in allr)})
    write_csv(os.path.join(out, 'range_summary.csv'), list(rsum[0].keys()) if rsum else ['dataset'], rsum)
    # the paired increment of the UK sweep (plan section 7): within each process, (T(k) - T(1)) / (K(k) - K(1)) with T the
    # process mean ns/request and K the mean returned keys/request; then median / min / max over processes
    inc = []
    pr = {}
    for d in rrounds:
        if d['paired'] != '1':
            continue
        pr.setdefault((d['dataset'], d['impl'], d['mode'], d['process']), {}).setdefault(int(d['k']), []).append(d)
    for (ds, impl, mode, p), byk in pr.items():
        if 1 not in byk:
            continue
        t1 = statistics.mean(float(x['ns_per_request']) for x in byk[1]); k1 = statistics.mean(float(x['keys']) / float(x['requests']) for x in byk[1])
        for kk, rounds_ in sorted(byk.items()):
            if kk == 1:
                continue
            tk = statistics.mean(float(x['ns_per_request']) for x in rounds_); kk_ret = statistics.mean(float(x['keys']) / float(x['requests']) for x in rounds_)
            inc.append({'dataset': ds, 'impl': impl, 'mode': mode, 'process': p, 'k': kk, 'T_k': tk, 'T_1': t1, 'K_k': kk_ret, 'K_1': k1,
                        'incremental_ns_per_additional_key': (tk - t1) / (kk_ret - k1) if kk_ret > k1 else ''})
    incsum = []
    g = {}
    for r in inc:
        g.setdefault((r['dataset'], r['impl'], r['mode'], r['k']), []).append(r['incremental_ns_per_additional_key'])
    for (ds, impl, mode, kk), v in sorted(g.items()):
        m = med3([x for x in v if x != ''])
        incsum.append({'dataset': ds, 'impl': impl, 'mode': mode, 'k': kk, 'n_processes': m[3], 'median': m[0], 'min': m[1], 'max': m[2]})
    write_csv(os.path.join(out, 'range_paired_increment.csv'), ['dataset', 'impl', 'mode', 'process', 'k', 'T_k', 'T_1', 'K_k', 'K_1', 'incremental_ns_per_additional_key'], inc)
    write_csv(os.path.join(out, 'range_paired_increment_summary.csv'), ['dataset', 'impl', 'mode', 'k', 'n_processes', 'median', 'min', 'max'], incsum)
    # ---- space (process 1), validation, calibration, profile
    for (ds, impl, p), r in sorted(accepted.items()):
        if r['space']:
            s = r['space']
            n = float(s['keys']); kb = float(r['load']['bytes']) - float(r['load']['n']) if r['load'] else 0   # canonical bytes minus one newline per key
            space.append({'dataset': ds, 'impl': impl, 'process': p, 'keys': s['keys'], 'standalone_bytes': s['standalone_bytes'],
                          'structure_bytes': s['structure_bytes'], 'backing_key_bytes': s['backing_key_bytes'], 'index_only_bytes': s['index_only_bytes'],
                          'bytes_per_key': float(s['standalone_bytes']) / n, 'bytes_per_input_key_byte': float(s['standalone_bytes']) / kb if kb else '',
                          'input_key_bytes': int(kb), 'basis': s['basis']})
            for c in r['components']:
                comps.append({'dataset': ds, 'impl': impl, 'process': p, 'name': c['name'], 'bytes': c['bytes'], 'in_total': c['in_total']})
        for d in r['validate']:
            validate.append({'dataset': ds, 'impl': impl, 'process': p, 'check': d.get('check', ''), 'total': d.get('total', ''), 'bad': d.get('bad', ''),
                             'verdict': d.get('status', d['verdict'])})
        for d in r['calibration']:
            calib.append({'dataset': ds, 'impl': impl, 'process': p, 'key': d['key'], 'batch_ns': d['batch_ns'], 'L': d['L'], 'round_ms_at_L': d['round_ms_at_L']})
        for d in r['profile']:
            profile.append({'dataset': ds, 'impl': impl, 'process': p, 'key': d['key'], 'phase': d['phase'], 'rounds': d['rounds'], 'wall_ns': d['wall_ns'],
                            'perf_enable': d['perf_enable'], 'perf_disable': d['perf_disable']})
    write_csv(os.path.join(out, 'space.csv'), ['dataset', 'impl', 'process', 'keys', 'standalone_bytes', 'structure_bytes', 'backing_key_bytes', 'index_only_bytes',
                                               'bytes_per_key', 'bytes_per_input_key_byte', 'input_key_bytes', 'basis'], space)
    write_csv(os.path.join(out, 'space_components.csv'), ['dataset', 'impl', 'process', 'name', 'bytes', 'in_total'], comps)
    write_csv(os.path.join(out, 'validate.csv'), ['dataset', 'impl', 'process', 'check', 'total', 'bad', 'verdict'], validate)
    write_csv(os.path.join(out, 'calibration.csv'), ['dataset', 'impl', 'process', 'key', 'batch_ns', 'L', 'round_ms_at_L'], calib)
    write_csv(os.path.join(out, 'profile.csv'), ['dataset', 'impl', 'process', 'key', 'phase', 'rounds', 'wall_ns', 'perf_enable', 'perf_disable'], profile)
    write_csv(os.path.join(out, 'attempts.csv'), ['dataset', 'name', 'kind', 'ident', 'attempt', 'rc', 'start', 'end', 'verdict', 'reasons'], attempts)
    # ---- coverage: the expected matrix (from the specification when one is given)
    cov = []
    if spec:
        import campaign_spec
        cov = campaign_spec.coverage(spec, accepted, statuses, rejected, violations)
        write_csv(os.path.join(out, 'coverage.csv'), ['dataset', 'impl', 'cell', 'status', 'accepted_processes', 'note'], cov)
        n_acc = sum(1 for c in cov if c['status'] == 'ACCEPTED'); n_req = sum(1 for c in cov if c['status'] not in ('OUT_OF_SCOPE', 'UNSUPPORTED_API'))
        print(f"SUMMARY_FULL spec={spec.get('name', '-')} accepted_processes={len(accepted)} spec_violations={len(violations)} "
              f"rejected={len(rejected)} coverage_accepted={n_acc}/{n_req} required cells (of {len(cov)} rows) out={out}")
        try:
            figures(out, bsum, psum, rsum, space)
        except Exception as e:
            print(f'FIGURES_SKIPPED {type(e).__name__}: {e}')
        return
    seen_ds = {ds for (ds, _, _) in list(accepted) + list(rejected)}
    for ds in DATASETS:
        if ds not in seen_ds:
            continue
        seen_sizes = sorted({d['size'] for d in prounds if d['dataset'] == ds}, key=lambda z: UK_SIZES.index(z) if z in UK_SIZES else 99)
        sizes = UK_SIZES if (ds == 'uk_2014_urls' or len(seen_sizes) == 5) else ['pct10']   # the five-size grid where it ran
        # the expected k list = the union of the range_ks DECLARED by the dataset's session plans (a paired single-k
        # session expects only its k; plan of October 30 section 4 item 4), else the archived k = 100
        declared = sorted({int(k) for (d2, i2, p2), r2 in accepted.items() if d2 == ds
                           for k in (r2['plan'].get('range_ks') or '-').split(',') if k not in ('-', '')})
        ks_expected = declared or [100]
        for impl in ARMS:
            procs = {p: accepted.get((ds, impl, p)) for p in (1, 2, 3)}
            acc = [p for p, r in procs.items() if r]
            # processes whose plan excluded range (plan 8.2: the DNA DT replacement runs build / point / census only)
            no_range = [p for p in acc if procs[p]['plan'].get('no_range') == '1']
            st = {p: statuses.get((ds, impl, p), '') for p in (1, 2, 3)}
            limit = next((s for s in st.values() if s in ('SIZE_LIMIT', 'UNSUPPORTED_KEY_DOMAIN', 'RESOURCE_LIMIT', 'TIME_LIMIT')), '')
            rej = [p for p in (1, 2, 3) if (ds, impl, p) in rejected and not procs[p]]

            def row(cell, supported, have, need=3, note=''):   # census and profile are process-1 records by design (need = 1)
                status = ('OUT_OF_SCOPE' if supported == 'scope' else 'UNSUPPORTED_API' if not supported else limit if limit else
                          'ACCEPTED' if len(have) >= need else 'PARTIAL' if have else 'REJECTED' if rej else 'MISSING')
                cov.append({'dataset': ds, 'impl': impl, 'cell': cell, 'status': status, 'accepted_processes': ''.join(str(p) for p in have),
                            'rejected_processes': ''.join(str(p) for p in rej), 'cell_end': ';'.join(f'p{p}={s}' for p, s in st.items() if s), 'note': note})
            row('build', True, acc)
            row('census', True, [1] if procs[1] and procs[1]['space'] else [], need=1)
            for z in sizes:
                for mode in ('direct', 'shared'):
                    sup = mode == 'direct' or impl in SHARED
                    key = f'point:hit:{mode}:{z}'
                    row(f'point:{z}:{mode}', sup, [p for p in acc if key in procs[p]['cells_seen']])
            for kk in ks_expected:
                for mode in ('direct', 'shared'):
                    sup = impl in RANGE and (mode == 'direct' or impl in SHARED)
                    key = f'range:{kk}:{mode}'
                    have = [p for p in acc if key in procs[p]['cells_seen']]
                    need = 3 - len(no_range) if sup and no_range else 3
                    note = f"range excluded by plan 8.2 in process {''.join(str(p) for p in no_range)}: {need} of 3" if sup and no_range else ''
                    row(f'range:k{kk}:{mode}', sup, have, need=need, note=note)
            # profiling: the k = 100 visits of XML / Wikipedia / WebBase / DNA (historical evidence); out of scope on UK and
            # for any process launched after the amendment of October 29 20:30 EDT (plan section 7)
            p1_no_profile = procs[1] is not None and procs[1]['plan'].get('profile_rounds', '0') == '0'
            row('profile', 'scope' if (ds == 'uk_2014_urls' or p1_no_profile) else impl in RANGE, [1] if procs[1] and procs[1]['profile'] else [], need=1)
    write_csv(os.path.join(out, 'coverage.csv'), ['dataset', 'impl', 'cell', 'status', 'accepted_processes', 'rejected_processes', 'cell_end', 'note'], cov)
    n_acc = sum(1 for c in cov if c['status'] == 'ACCEPTED'); n_all = len(cov)
    limits = sum(1 for k in rejected if statuses.get(k, '') in ('SIZE_LIMIT', 'UNSUPPORTED_KEY_DOMAIN', 'RESOURCE_LIMIT', 'TIME_LIMIT'))
    print(f'SUMMARY_FULL datasets={len(seen_ds)} accepted_processes={len(accepted)} recorded_limits={limits} rejected={len(rejected) - limits} '
          f'coverage_accepted={n_acc}/{n_all} out={out}')
    try:
        figures(out, bsum, psum, rsum, space)
    except Exception as e:   # the tables are the deliverable; figures are best effort here (plot_full.py redraws them)
        print(f'FIGURES_SKIPPED {type(e).__name__}: {e}')


def figures(out, bsum, psum, rsum, space):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fdir = os.path.join(out, 'figures')
    os.makedirs(fdir, exist_ok=True)
    arms = [a for a in ARMS]
    colors = {a: c for a, c in zip(arms, plt.rcParams['axes.prop_cycle'].by_key()['color'])}

    def grouped(ax, rows, val, lo, hi, ylabel, title, log=False, modes=None):
        dss = [d for d in DATASETS if any(r['dataset'] == d for r in rows)]
        series = [(a, m) for a in arms for m in (modes or [None])]
        width = 0.8 / max(1, len(series))
        for i, (a, m) in enumerate(series):
            xs, ys, err = [], [], [[], []]
            for j, d in enumerate(dss):
                r = next((r for r in rows if r['dataset'] == d and r['impl'] == a and (m is None or r.get('mode') == m)), None)
                if not r or r[val] == '':
                    continue
                xs.append(j + i * width); ys.append(float(r[val]))
                err[0].append(float(r[val]) - float(r[lo])); err[1].append(float(r[hi]) - float(r[val]))
            if xs:
                ax.bar(xs, ys, width, yerr=err, capsize=2, label=LABEL[a] + (f' {m}' if m else ''), color=colors[a], alpha=1.0 if m in (None, 'direct') else 0.5)
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(dss))]); ax.set_xticklabels([LABEL[d] for d in dss])
        ax.set_ylabel(ylabel); ax.set_title(title)
        if log:
            ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=2)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    grouped(ax, bsum, 'e2e_median_s', 'e2e_min_s', 'e2e_max_s', 'seconds (log)',
            'Construction, end-to-end from the shuffled list (median of 3 single builds; whiskers min-max; 64 workers where the builder is parallel)', log=True)
    fig.tight_layout(); fig.savefig(os.path.join(fdir, 'build_time.png'), dpi=150); plt.close(fig)
    if space:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sp = [dict(r, lo=r['bytes_per_key'], hi=r['bytes_per_key']) for r in space]
        grouped(ax, sp, 'bytes_per_key', 'lo', 'hi', 'bytes per key', 'Serving space of the retained process-1 index (each arm\'s accounting basis; labelled records)')
        fig.tight_layout(); fig.savefig(os.path.join(fdir, 'serving_space.png'), dpi=150); plt.close(fig)
    p10 = [r for r in psum if r['size'] == 'pct10' and r['dataset'] != 'uk_2014_urls']
    if p10:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        grouped(ax, p10, 'median_ns_per_query', 'min_ns_per_query', 'max_ns_per_query', 'ns per query (64 workers, elapsed / queries)',
                'Hit point queries, batch = 10 % of N (median of 3 process means; whiskers min-max)', modes=['direct', 'shared'])
        fig.tight_layout(); fig.savefig(os.path.join(fdir, 'point_10pct.png'), dpi=150); plt.close(fig)
    for ds_sw in DATASETS:   # the five-size point sweep figure of every dataset that has it (UK-2014; the WebBase repeat)
      uk = [r for r in psum if r['dataset'] == ds_sw]
      if len({r['size'] for r in uk}) < 5:
        continue
      if True:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for a in arms:
            for m in ('direct', 'shared'):
                rows = [r for r in uk if r['impl'] == a and r['mode'] == m]
                rows.sort(key=lambda r: UK_SIZES.index(r['size']))
                if rows:
                    ax.errorbar([UK_SIZES.index(r['size']) for r in rows], [float(r['median_ns_per_query']) for r in rows],
                                yerr=[[float(r['median_ns_per_query']) - float(r['min_ns_per_query']) for r in rows],
                                      [float(r['max_ns_per_query']) - float(r['median_ns_per_query']) for r in rows]],
                                label=LABEL[a] + f' {m}', color=colors[a], linestyle='-' if m == 'direct' else '--', marker='o', capsize=2)
        ax.set_xticks(range(5)); ax.set_xticklabels(['0.001 %', '0.01 %', '0.1 %', '1 %', '10 %']); ax.set_xlabel('point batch size (fraction of N)')
        ax.set_ylabel('ns per query'); ax.set_yscale('log'); ax.set_title(f'{LABEL[ds_sw]} hit point queries by batch size (median of 3 process means)')
        ax.legend(fontsize=7, ncol=2); fig.tight_layout()
        fig.savefig(os.path.join(fdir, 'point_uk_sweep.png' if ds_sw == 'uk_2014_urls' else f'point_sweep_{ds_sw}.png'), dpi=150); plt.close(fig)
    r100 = [r for r in rsum if r['k'] == '100' and r['paired'] != '1']
    if r100:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        grouped(ax, r100, 'median_ns_per_request', 'min_ns_per_request', 'max_ns_per_request', 'ns per range request (64 workers)',
                'ARCHIVED (full-pop-hit-v1): full-key range queries, k = 100, B = 40,000, hit lower bounds (median of 3 process means; whiskers min-max)',
                modes=['direct', 'shared'])
        fig.tight_layout(); fig.savefig(os.path.join(fdir, 'range_k100_archived.png'), dpi=150); plt.close(fig)
    for ds_sw in DATASETS:
      sweep = [r for r in rsum if r['paired'] == '1' and r['dataset'] == ds_sw]
      if sweep:   # the primary range figure (plan section 7): x = k (log), y = ns/request, direct and shared panels
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, mode in zip(axes, ('direct', 'shared')):
            for a in arms:
                rows = sorted([r for r in sweep if r['impl'] == a and r['mode'] == mode], key=lambda r: int(r['k']))
                if rows:
                    ax.errorbar([int(r['k']) for r in rows], [float(r['median_ns_per_request']) for r in rows],
                                yerr=[[float(r['median_ns_per_request']) - float(r['min_ns_per_request']) for r in rows],
                                      [float(r['max_ns_per_request']) - float(r['median_ns_per_request']) for r in rows]],
                                label=LABEL[a], color=colors[a], marker='o', capsize=2)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([1, 10, 100, 1000, 10000]); ax.set_xticklabels(['1', '10', '100', '1000', '10000'])
            ax.set_xlabel('k (requested keys per range request)'); ax.set_title(f'{mode}')
            ax.legend(fontsize=8)
        axes[0].set_ylabel('elapsed ns per range request (64 workers)')
        fig.suptitle(f'{LABEL[ds_sw]}, full population, hit-only lower bounds, complete-key output, B = 40,000 requests per batch; '
                     'median of 3 process means (one build each, 2 warm-up + 4 measured rounds per cell), whiskers min-max; throughput-derived cost', fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, 'range_uk_sweep.png' if ds_sw == 'uk_2014_urls' else f'range_sweep_{ds_sw}.png'), dpi=150); plt.close(fig)
    print('FIGURES', fdir)


if __name__ == '__main__':
    main()
