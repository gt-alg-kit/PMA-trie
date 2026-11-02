#!/usr/bin/env python3
"""Preparation of the END/B DFUDS / DT rerun (docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md; runs ON THE MAC, nothing is
launched). From the plan's constants and the historical records it produces, into <bundle-dir> and into
<tree-dir> (default: rerun_2026-11-01/ next to this script, so that the frozen snapshot carries the same files):

  specs/end_b_full_hit_2026-11-01.json       Part A campaign specification (campaign_spec.py; per-dataset point sizes / k lists,
                                             query_modes all + range_modes direct, supplied L, key-end policies, historical
                                             paired-trace hashes; binaries filled at launch from FROZEN.txt)
  specs/end_b_insertion_2026-11-01.json      Part B specification (insert_spec.py; the 22 cells, split / batch identity of the
                                             October 24 insertion logs, historical log hashes)
  L_manifests/<dataset>/L_manifest.tsv       the selected inherited L values (plan section 4), source path + SHA-256 per row
  L_manifests/SHA256SUMS, L_manifest_sources.json
  previews/full_hit_<dataset>.tsv, previews/insertion_<dataset>.tsv   the exact schedules (placeholder root / results paths)
  coverage_expected.md / .csv, counts.json  the declared matrix and the plan's counts, checked (PREPARE_CHECK lines)
  plan_constants.json, time_budget.md, historical_identity.json
usage: rerun_prepare.py <repo-root> <bundle-dir> [--tree-dir D] [--root-placeholder P] [--results-placeholder P]
Exit status 0 only when every check passed.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign_spec   # noqa: E402
import insert_spec     # noqa: E402

CAMPAIGN = 'end-b-full-hit-20261101-v1'
SALT_A = 'end-b-full-hit-20261101-v1'
SALT_B = 'end-b-insertion-20261101-v1'
DATASETS = ['c2_xml_recon', 'enwiki_20261001_ns0', 'webbase_2001', 'dna_full_31mers', 'uk_2014_urls']
N = {'c2_xml_recon': 2149271, 'enwiki_20261001_ns0': 19315921, 'webbase_2001': 117603792, 'dna_full_31mers': 367422517, 'uk_2014_urls': 787801471}
ARMS = ['dfuds', 'dfuds-dt']
FIVE = ['pct0.001', 'pct0.01', 'pct0.1', 'pct1', 'pct10']
POINT_SIZES = {ds: (FIVE if ds == 'webbase_2001' else ['pct10']) for ds in DATASETS}
RANGE_KS = {ds: ([1, 10, 100, 1000, 10000] if ds == 'webbase_2001' else [100]) for ds in DATASETS}
# Part B (plan section 9): DFUDS 0.001 % and 10 % everywhere, DT 10 % everywhere, both arms every size on WebBase
INSERT_PCTS = {ds: ({'dfuds': ['0.001', '0.01', '0.1', '1', '10'], 'dfuds-dt': ['0.001', '0.01', '0.1', '1', '10']} if ds == 'webbase_2001'
                    else {'dfuds': ['0.001', '10'], 'dfuds-dt': ['10']}) for ds in DATASETS}
KEY_END = {'dfuds': 'end_leaf', 'dfuds-dt': 'end_leaf_front_and_rear'}
# the inherited L values of plan section 4 (cross-checked against the selected manifest rows)
EXPECTED_L = {
    ('c2_xml_recon', 'point:hit:direct:pct10'): 21, ('c2_xml_recon', 'point:hit:shared:pct10'): 43, ('c2_xml_recon', 'range:100'): 31,
    ('enwiki_20261001_ns0', 'point:hit:direct:pct10'): 3, ('enwiki_20261001_ns0', 'point:hit:shared:pct10'): 7, ('enwiki_20261001_ns0', 'range:100'): 32,
    ('webbase_2001', 'point:hit:direct:pct0.001'): 1000, ('webbase_2001', 'point:hit:shared:pct0.001'): 239,
    ('webbase_2001', 'point:hit:direct:pct0.01'): 188, ('webbase_2001', 'point:hit:shared:pct0.01'): 69,
    ('webbase_2001', 'point:hit:direct:pct0.1'): 20, ('webbase_2001', 'point:hit:shared:pct0.1'): 28,
    ('webbase_2001', 'point:hit:direct:pct1'): 2, ('webbase_2001', 'point:hit:shared:pct1'): 7,
    ('webbase_2001', 'point:hit:direct:pct10'): 1, ('webbase_2001', 'point:hit:shared:pct10'): 2,
    ('webbase_2001', 'range:1'): 51, ('webbase_2001', 'range:10'): 50, ('webbase_2001', 'range:100'): 26, ('webbase_2001', 'range:1000'): 5, ('webbase_2001', 'range:10000'): 1,
    ('dna_full_31mers', 'point:hit:direct:pct10'): 1, ('dna_full_31mers', 'point:hit:shared:pct10'): 1, ('dna_full_31mers', 'range:100'): 23,
    ('uk_2014_urls', 'point:hit:direct:pct10'): 1, ('uk_2014_urls', 'point:hit:shared:pct10'): 1, ('uk_2014_urls', 'range:100'): 19,
}
FULL_HIT = 'results/dictionary_comparison_full_hit_2026-10-29/remote'
K100 = 'results/dictionary_comparison_k100_webbase_2026-10-30'
MAIN = 'results/dictionary_comparison_2026-10-23/main_campaign/remote'


def l_sources():
    """(dataset, key) -> (manifest path, key in that manifest): plan section 4."""
    src = {}
    for ds in DATASETS:
        if ds == 'webbase_2001':
            m = f'{K100}/task_a/remote/webbase_2001/L_manifest.tsv'
            for z in FIVE:
                for mode in ('direct', 'shared'):
                    src[(ds, f'point:hit:{mode}:{z}')] = (m, f'point:hit:{mode}:{z}')
            for k in RANGE_KS[ds]:
                src[(ds, f'range:{k}')] = (m, f'range:{k}')
            continue
        for mode in ('direct', 'shared'):
            src[(ds, f'point:hit:{mode}:pct10')] = (f'{FULL_HIT}/{ds}/L_manifest.tsv', f'point:hit:{mode}:pct10')
        if ds == 'uk_2014_urls':
            src[(ds, 'range:100')] = (f'{FULL_HIT}/uk_2014_urls/L_manifest.tsv', 'range:100')
        else:
            src[(ds, 'range:100')] = (f'{K100}/task_b/remote/{ds}/L_manifest.tsv', 'range:100')
    return src


TRACE_SRC = {'c2_xml_recon': f'{K100}/task_b/remote/c2_xml_recon/sessions/raw/S__c2_xml_recon__dfuds__p1.log',
             'enwiki_20261001_ns0': f'{K100}/task_b/remote/enwiki_20261001_ns0/sessions/raw/S__enwiki_20261001_ns0__dfuds__p1.log',
             'webbase_2001': f'{K100}/task_a/remote/webbase_2001/sessions/raw/S__webbase_2001__dfuds__p1.log',
             'dna_full_31mers': f'{K100}/task_b/remote/dna_full_31mers/sessions/raw/S__dna_full_31mers__dfuds__p1.log',
             'uk_2014_urls': f'{FULL_HIT}/uk_2014_urls/sessions/raw/S__uk_2014_urls__dfuds__p1.log'}
HIST_RANGE_HASH_WEBBASE = '2d1d6a1358caaf43'   # plan section 4
HIST_MIN = {'Part A': {'c2_xml_recon': 3.5, 'enwiki_20261001_ns0': 4.5, 'webbase_2001': 17.6, 'dna_full_31mers': 19.9, 'uk_2014_urls': 56.3},
            'Part B': {'c2_xml_recon': 0.8, 'enwiki_20261001_ns0': 1.3, 'webbase_2001': 12.4, 'dna_full_31mers': 28.8, 'uk_2014_urls': 54.4}}
checks = []


def check(name, ok, detail=''):
    checks.append((name, bool(ok), detail))
    print(f"PREPARE_CHECK {name} {'PASS' if ok else 'FAIL'} {detail}")
    return ok


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_manifest_rows(path):
    rows = {}
    for line in open(path):
        line = line.rstrip('\n')
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        rows[parts[0]] = (int(parts[1]), line)   # the last line of a key wins (driver rule)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo')
    ap.add_argument('bundle')
    ap.add_argument('--tree-dir', default=os.path.join(HERE, 'rerun_2026-11-01'))
    ap.add_argument('--root-placeholder', default='/REMOTE/dictcmp/main-TAG')
    ap.add_argument('--results-placeholder', default='/REMOTE/dictcmp/results_end_b_full_hit_2026-11-01')
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    outs = [os.path.abspath(a.bundle), os.path.abspath(a.tree_dir)]
    for o in outs:
        for d in ('specs', 'L_manifests', 'previews'):
            os.makedirs(os.path.join(o, d), exist_ok=True)
    # ---- 1. the selected L manifests
    src = l_sources()
    sources_rec, manifests, sums = {}, {}, []
    for ds in DATASETS:
        lines, rec = [], []
        keys = [f'point:hit:{m}:{z}' for z in POINT_SIZES[ds] for m in ('direct', 'shared')] + [f'range:{k}' for k in RANGE_KS[ds]]
        for key in keys:
            mpath, mkey = src[(ds, key)]
            full = os.path.join(repo, mpath)
            rows = read_manifest_rows(full)
            if mkey not in rows:
                check(f'L_source_{ds}_{key}', False, f'{mpath} lacks {mkey}')
                continue
            L, raw = rows[mkey]
            exp = EXPECTED_L.get((ds, key))
            check(f'L_value_{ds}_{key}', exp == L, f'manifest {L} plan {exp} ({mpath})')
            h = sha256(full)
            orig = raw.split('\t', 2)[2] if raw.count('\t') >= 2 else ''
            lines.append(f'{key}\t{L}\tinherited {mpath} sha256={h[:16]} | {orig}')
            rec.append({'key': key, 'L': L, 'source_file': mpath, 'source_sha256': h, 'source_row': raw, 'source_key': mkey})
        text = '# selected inherited L values of the END/B rerun (plan section 4): key, L, provenance; not recalibrated\n' + '\n'.join(lines) + '\n'
        manifests[ds] = text
        sources_rec[ds] = rec
        for o in outs:
            d = os.path.join(o, 'L_manifests', ds)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, 'L_manifest.tsv'), 'w').write(text)
        sums.append(f'{hashlib.sha256(text.encode()).hexdigest()}  {ds}/L_manifest.tsv')
        check(f'L_manifest_{ds}_keys', len(lines) == len(keys) and len({l.split(chr(9))[0] for l in lines}) == len(keys), f'{len(lines)} rows')
    for o in outs:
        open(os.path.join(o, 'L_manifests', 'SHA256SUMS'), 'w').write('\n'.join(sums) + '\n')
        json.dump(sources_rec, open(os.path.join(o, 'L_manifest_sources.json'), 'w'), indent=1)
    # ---- 2. historical paired-trace hashes
    traces = {}
    for ds, rel in TRACE_SRC.items():
        full = os.path.join(repo, rel)
        line = next((l for l in open(full, errors='replace') if l.startswith('RANGE_PAIRED_TRACE ')), '')
        m = re.search(r'starts_ids_hash=([0-9a-f]+) starts_bytes_hash=([0-9a-f]+) identical=1', line)
        if m:
            traces[ds] = {'starts_ids_hash': m.group(1), 'starts_bytes_hash': m.group(2), 'source_log': rel, 'source_sha256': sha256(full)}
        check(f'trace_hash_{ds}', bool(m), f"{traces.get(ds, {}).get('starts_ids_hash')} from {rel}")
    check('trace_hash_webbase_plan', traces.get('webbase_2001', {}).get('starts_ids_hash') == HIST_RANGE_HASH_WEBBASE, HIST_RANGE_HASH_WEBBASE)
    # ---- 3. Part A specification
    spec_a = {'name': CAMPAIGN, 'processes': 3, 'workers': 64, 'cpus': '0-63', 'population': 'full', 'point_sizes': ['pct10'],
              'query_modes': 'all', 'range_modes': 'direct', 'range_ks': [100], 'range_paired': True, 'range_batch': 40000,
              'range_workloads': ['full-pop-range-paired-v1'], 'census_process': 1, 'profile': False, 'warm_rounds': 2, 'measured_rounds': 4,
              'datasets': DATASETS, 'arms': ARMS,
              'per_dataset': {'webbase_2001': {'point_sizes': FIVE, 'range_ks': [1, 10, 100, 1000, 10000]}},
              'supplied_L': True, 'key_end': KEY_END,
              'expected_trace_hashes': {ds: {k: v for k, v in t.items() if k.startswith('starts')} for ds, t in traces.items()},
              'binaries': {'bench_dictcmp': ['TO_BE_FILLED_FROM_FROZEN_TXT'], 'bench_dictcmp_fst64': []},
              'salt': SALT_A, 'experiment': 'full-pop-hit-v1', 'range_experiment': 'full-pop-range-paired-v1',
              'source_root': '~/dictcmp/results_end_b_full_hit_2026-11-01/full_hit', 'snapshot': '~/dictcmp/main-<tag> (frozen by freeze_rerun.sh)',
              'plan': 'docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md', 'prepared': 'rerun_prepare.py'}
    for o in outs:
        json.dump(spec_a, open(os.path.join(o, 'specs', 'end_b_full_hit_2026-11-01.json'), 'w'), indent=1)
    loaded_a = campaign_spec.load(os.path.join(outs[0], 'specs', 'end_b_full_hit_2026-11-01.json'))
    check('spec_a_loads', loaded_a['range_modes'] == 'direct' and loaded_a['query_modes'] == 'all')
    # ---- 4. Part B specification from the historical insertion logs
    datasets_b, identity, hist_logs, hist_bins = {}, {}, {}, set()
    for ds in DATASETS:
        identity[ds] = {}
        for pct in ['0.001', '0.01', '0.1', '1', '10']:
            recs = []
            for impl in ARMS:
                for p in (1, 2, 3):
                    rel = f'{MAIN}/{ds}/measure/raw/I__{ds}__{impl}__pct{pct}__p{p}.log'
                    full = os.path.join(repo, rel)
                    if not os.path.exists(full):
                        continue
                    r = insert_spec.parse_log(full)
                    recs.append((rel, r))
                    hist_logs[rel] = sha256(full)
                    hist_bins.add(r['cell_run'].get('binary_sha'))
            ids = {(tuple(r['ids'].get(t) for t in range(3)), r['plan'].get('b'), r['plan'].get('final'), r['split'].get('perm_hash'),
                    r['split'].get('stored_sorted_hash'), r['split'].get('S'), r['split'].get('H'), r['split'].get('N')) for _, r in recs}
            check(f'hist_identity_{ds}_pct{pct}', len(recs) == 6 and len(ids) == 1, f'{len(recs)} logs, {len(ids)} distinct identities')
            if len(ids) != 1:
                continue
            (hashes, b, final, perm, ssh_, S, H, Nn) = next(iter(ids))
            identity[ds][pct] = {'b': int(b), 'final': int(final), 'ids_hash': list(hashes)}
            datasets_b[ds] = {'N': int(Nn), 'S': int(S), 'H': int(H), 'perm_hash': perm, 'stored_sorted_hash': ssh_}
        check(f'hist_split_{ds}', datasets_b.get(ds, {}).get('N') == N[ds] and datasets_b.get(ds, {}).get('S') == 7 * N[ds] // 10, str(datasets_b.get(ds)))
    cells = [{'dataset': ds, 'impl': impl, 'pct': pct} for ds in DATASETS for impl in ARMS for pct in INSERT_PCTS[ds][impl]]
    spec_b = {'name': SALT_B, 'salt': SALT_B, 'workers': 64, 'cpus': '0-63', 'population': 'split70', 'processes': 3, 'warm_rounds': 2,
              'measured_rounds': 4, 'batches': 3, 'method': 'batch_sorted_in_timer', 'seed': 20260928, 'arms': ARMS, 'cells': cells,
              'datasets': datasets_b, 'identity': identity, 'key_end': KEY_END,
              'binaries': {'bench_dictcmp': ['TO_BE_FILLED_FROM_FROZEN_TXT']},
              'insert_flags': '--population split70 --warm-rounds 2 --rounds 4',
              'historical': {'root': MAIN, 'binary_sha16': sorted(hist_bins), 'logs_sha256': hist_logs,
                             'note': 'identity values (permutation, initial set, per-batch ID hashes) of the accepted October 24-26 insertion '
                                     'processes; their timings are the retained old observations, not part of this campaign'},
              'source_root': '~/dictcmp/results_end_b_full_hit_2026-11-01/insertion', 'plan': 'docs/dfuds_dt_full_hit_rerun_plan_2026-11-01.md section 9'}
    for o in outs:
        json.dump(spec_b, open(os.path.join(o, 'specs', 'end_b_insertion_2026-11-01.json'), 'w'), indent=1)
    loaded_b = insert_spec.load(os.path.join(outs[0], 'specs', 'end_b_insertion_2026-11-01.json'))
    check('spec_b_loads_22_cells', len(loaded_b['cells']) == 22, f"{len(loaded_b['cells'])} cells")
    check('hist_binary_single', hist_bins == {'4ae0d3084f1c4d2b'}, str(sorted(hist_bins)))
    # ---- 5. schedule previews (placeholders) and their checks
    R, RES = a.root_placeholder, a.results_placeholder
    launch = f'{R}/src/trie/bench/protocols/dictcmp/launch.sh'
    for ds in DATASETS:
        cmd = [sys.executable, os.path.join(HERE, 'schedule_full.py'), '--dataset', ds, '--bin', f'{R}/build-rel/bench_dictcmp',
               '--fst64-bin', f'{R}/build-rel/bench_dictcmp_fst64', '--launch', launch, '--L-manifest', f'{RES}/full_hit/{ds}/L_manifest.tsv',
               '--perf-dir', f'{RES}/full_hit/{ds}/perf', '--procs', '3', '--impls', ','.join(ARMS), '--salt', SALT_A, '--range-policy', 'sweep',
               '--range-ks', ','.join(str(k) for k in RANGE_KS[ds]), '--range-experiment', 'full-pop-range-paired-v1',
               '--point-sizes', ','.join(POINT_SIZES[ds]), '--profile-rounds', '0', '--session-range-modes', 'direct']
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        rows = [l.split('\t') for l in out.strip().split('\n')]
        for o in outs:
            open(os.path.join(o, 'previews', f'full_hit_{ds}.tsv'), 'w').write(out)
        names = [r[0] for r in rows]
        exp_names = [f'S__{ds}__{impl}__p{p}' for p in (1, 2, 3) for impl in (ARMS if p != 2 else ARMS[::-1])]
        ok = names == exp_names and all(r[1] == 'SESSION' and r[2] == SALT_A for r in rows)
        ok &= all('--session-range-modes direct' in r[3] and '--range-paired' in r[3] and '--population full' in r[3] and '--profile-rounds' not in r[3]
                  and '--session-query-modes' not in r[3] and '--range-batch 40000' in r[3] for r in rows)
        ok &= sum('--census' in r[3] for r in rows) == 2 and all(('--census' in r[3]) == ('__p1' in r[0]) for r in rows)
        check(f'schedule_A_{ds}', ok, f'{len(rows)} rows, order {[n.split("__")[2] + n[-3:] for n in names]}')
        keys = {f'point:hit:{m}:{z}' for z in POINT_SIZES[ds] for m in ('direct', 'shared')} | {f'range:{k}' for k in RANGE_KS[ds]}
        have = {l.split('\t')[0] for l in manifests[ds].split('\n') if l and not l.startswith('#')}
        check(f'L_covers_schedule_{ds}', keys <= have, f'missing {sorted(keys - have)}')
        cmd = [sys.executable, os.path.join(HERE, 'schedule.py'), 'measure', '--dataset', ds, '--bin', f'{R}/build-rel/bench_dictcmp',
               '--census-bin', f'{R}/build-rel/bench_dictcmp', '--launch', launch, '--procs', '3', '--impls', ','.join(ARMS), '--panels', 'insert',
               '--insert-flags', spec_b['insert_flags'], '--salt', SALT_B]
        for impl in ARMS:
            cmd += ['--pcts', f'{impl}:{",".join(INSERT_PCTS[ds][impl])}']
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        rows = [l.split('\t') for l in out.strip().split('\n')]
        for o in outs:
            open(os.path.join(o, 'previews', f'insertion_{ds}.tsv'), 'w').write(out)
        want = sorted(f'I__{ds}__{c["impl"]}__pct{c["pct"]}__p{p}' for c in cells if c['dataset'] == ds for p in (1, 2, 3))
        ok = sorted(r[0] for r in rows) == want and all(r[1] == 'INSERT' and r[2] == SALT_B and '--mode insert' in r[3]
                                                         and spec_b['insert_flags'] in r[3] for r in rows)
        check(f'schedule_B_{ds}', ok, f'{len(rows)} rows (expected {len(want)})')
    # ---- 6. coverage tables and the plan's counts
    cov_rows = []
    counts = {'sessions': 0, 'builds': 0, 'censuses': 0, 'point_aggregates': 0, 'range_aggregates': 0, 'query_process_observations': 0,
              'measured_query_rounds': 0, 'warm_up_query_rounds': 0, 'paired_increment_aggregates': 0}
    for ds in DATASETS:
        sd = campaign_spec.spec_for(loaded_a, ds)
        for impl in ARMS:
            counts['sessions'] += 3
            for cell, scope in campaign_spec.expected_cells(sd, impl):
                need = 1 if cell in ('census', 'profile') else 3
                cov_rows.append({'part': 'A', 'dataset': ds, 'impl': impl, 'cell': cell, 'scope': scope, 'processes': need if scope == 'required' else 0})
                if scope != 'required':
                    continue
                if cell == 'build':
                    counts['builds'] += 3
                elif cell == 'census':
                    counts['censuses'] += 1
                elif cell.startswith('point:'):
                    counts['point_aggregates'] += 1; counts['query_process_observations'] += 3; counts['measured_query_rounds'] += 12; counts['warm_up_query_rounds'] += 6
                elif cell.startswith('range:'):
                    counts['range_aggregates'] += 1; counts['query_process_observations'] += 3; counts['measured_query_rounds'] += 12; counts['warm_up_query_rounds'] += 6
                    if ds == 'webbase_2001' and not cell.startswith('range:k1:'):
                        counts['paired_increment_aggregates'] += 1
    for c in insert_spec.cells_of(loaded_b):
        cov_rows.append({'part': 'B', 'dataset': c['dataset'], 'impl': c['impl'], 'cell': f"insert:pct{c['pct']}:b{loaded_b['identity'][c['dataset']][c['pct']]['b']}",
                         'scope': 'required', 'processes': 3})
    nb = len(loaded_b['cells'])
    counts.update({'insertion_cells': nb, 'insertion_processes': 3 * nb, 'insertion_fresh_builds': 18 * nb, 'insertion_warm_up_rounds': 6 * nb,
                   'insertion_measured_rounds': 12 * nb, 'insertion_batch_records': 54 * nb, 'insertion_measured_batches': 36 * nb,
                   'insertion_headline_aggregates': nb, 'insertion_batch_position_aggregates': 3 * nb})
    expect = {'sessions': 30, 'builds': 30, 'censuses': 10, 'point_aggregates': 36, 'range_aggregates': 18, 'query_process_observations': 162,
              'measured_query_rounds': 648, 'warm_up_query_rounds': 324, 'paired_increment_aggregates': 8, 'insertion_cells': 22, 'insertion_processes': 66,
              'insertion_fresh_builds': 396, 'insertion_warm_up_rounds': 132, 'insertion_measured_rounds': 264, 'insertion_batch_records': 1188,
              'insertion_measured_batches': 792, 'insertion_headline_aggregates': 22, 'insertion_batch_position_aggregates': 66}
    for k, v in expect.items():
        check(f'count_{k}', counts[k] == v, f'{counts[k]} (plan {v})')
    counts['total_processes'] = counts['sessions'] + counts['insertion_processes']
    counts['total_index_constructions'] = counts['builds'] + counts['insertion_fresh_builds']
    check('count_total_processes_96', counts['total_processes'] == 96)
    check('count_total_constructions_426', counts['total_index_constructions'] == 426)
    LBL = insert_spec.LABEL
    L = ['# Expected coverage of the END/B rerun (declared before launch)\n',
         f'Campaign `{CAMPAIGN}`; Part A salt `{SALT_A}`, Part B salt `{SALT_B}`. Scope = required (measured, 3 processes unless noted), '
         'OUT_OF_SCOPE (the shared range cells: direct-only range by the plan) or UNSUPPORTED_API. Source: the two specifications in `specs/`.\n',
         '## Part A: full-population sessions (one build per process)\n',
         '| Dataset | Arm | Cells required (per process) | Shared range cells | Processes | Sessions |', '|---|---|---|---|---:|---:|']
    for ds in DATASETS:
        for impl in ARMS:
            ec = campaign_spec.expected_cells(campaign_spec.spec_for(loaded_a, ds), impl)
            req = [c for c, s in ec if s == 'required' and c not in ('build', 'census')]
            oos = [c for c, s in ec if s == 'OUT_OF_SCOPE']
            L.append(f"| {LBL[ds]} | {LBL[impl]} | build, census (p1), {', '.join(req)} | {', '.join(oos) or '-'} (OUT_OF_SCOPE) | 3 | 3 |")
    L += ['', '## Part B: insertion processes (70 / 30 split, fresh S-key build per round)\n',
          '| Dataset | Arm | Sizes (% of N) | b per size | Processes | Fresh builds | Measured batches |', '|---|---|---|---|---:|---:|---:|']
    for ds in DATASETS:
        for impl in ARMS:
            ps = INSERT_PCTS[ds][impl]
            bs = ', '.join(f"{loaded_b['identity'][ds][p]['b']:,}" for p in ps)
            L.append(f"| {LBL[ds]} | {LBL[impl]} | {', '.join(ps)} | {bs} | {3 * len(ps)} | {18 * len(ps)} | {36 * len(ps)} |")
    L += ['', '## Counts (checked against the plan)\n', '| Quantity | Value |', '|---|---:|'] + [f'| {k} | {v:,} |' for k, v in counts.items()]
    L += ['', 'Reused endpoints: the WebBase pct10 point (direct and shared) and k = 100 range observations of the sweep sessions supply the '
          'five-dataset overview; the WebBase 10 % insertion cells supply the Table 3 row and its 0.001 % DFUDS cell the endpoint claim. '
          'They are the same observations, counted once above.']
    T = ['# Time budget references (plan section 7; scheduling references, not END/B predictions)\n',
         '| Dataset | Part A historical session total (min) | Part B historical 66-process share (min) | Sum (min) |', '|---|---:|---:|---:|']
    ta = tb = 0
    for ds in DATASETS:
        ta += HIST_MIN['Part A'][ds]; tb += HIST_MIN['Part B'][ds]
        T.append(f"| {LBL[ds]} | {HIST_MIN['Part A'][ds]} | {HIST_MIN['Part B'][ds]} | {HIST_MIN['Part A'][ds] + HIST_MIN['Part B'][ds]:.1f} |")
    T += [f'| Total | {ta:.1f} | {tb:.1f} | {ta + tb:.1f} |', '',
          'Plan allowance: 1.5-2.5 h for Part A alone, 3.5-5 h for both parts after the freeze and machine availability; '
          'revise from accepted sessions (keep preparation, machine-wait and measurement elapsed times apart). '
          'Historical timed construction of the 30 full-population builds was about 4.6 minutes.']
    for o in outs:
        open(os.path.join(o, 'coverage_expected.md'), 'w').write('\n'.join(L) + '\n')
        with open(os.path.join(o, 'coverage_expected.csv'), 'w') as f:
            f.write('part,dataset,impl,cell,scope,processes\n')
            for r in cov_rows:
                f.write(','.join(str(r[k]) for k in ('part', 'dataset', 'impl', 'cell', 'scope', 'processes')) + '\n')
        json.dump(counts, open(os.path.join(o, 'counts.json'), 'w'), indent=1)
        json.dump({'campaign': CAMPAIGN, 'salts': {'A': SALT_A, 'B': SALT_B}, 'datasets': DATASETS, 'N': N, 'arms': ARMS, 'point_sizes': POINT_SIZES,
                   'range_ks': RANGE_KS, 'insert_pcts': INSERT_PCTS, 'insert_b': {ds: {p: identity[ds][p]['b'] for p in identity[ds]} for ds in DATASETS},
                   'split': datasets_b, 'key_end': KEY_END, 'experiment': 'full-pop-hit-v1', 'range_experiment': 'full-pop-range-paired-v1',
                   'range_batch': 40000, 'workers': 64, 'cpus': '0-63', 'warm_rounds': 2, 'measured_rounds': 4, 'processes': 3,
                   'order': 'XML, Wikipedia, WebBase, DNA, UK; per dataset Part A (6 sessions) then Part B'},
                  open(os.path.join(o, 'plan_constants.json'), 'w'), indent=1)
        json.dump({'paired_trace': traces, 'insertion': {'datasets': datasets_b, 'identity': identity, 'binary_sha16': sorted(hist_bins), 'logs_sha256': hist_logs}},
                  open(os.path.join(o, 'historical_identity.json'), 'w'), indent=1)
        open(os.path.join(o, 'time_budget.md'), 'w').write('\n'.join(T) + '\n')
    fails = sum(1 for _, ok, _ in checks if not ok)
    print(f'PREPARE_DONE checks={len(checks)} failures={fails} bundle={outs[0]} tree={outs[1]}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
